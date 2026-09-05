# Is the misfiling in folder 103 present in the other 37 folders?

**PRIVATE.** Names real titles and paths. Never copied into the repository.

Asked by the Lead after the baseline run on error id 237 turned up a complete
26-episode run of a different series inside one folder. If that family is really
8 % of the corpus it outranks everything else on the board — so the number has to
be right, and the method has to be one that can be wrong out loud.

Status: **answered.** See RESULTS at the end.

---

## Three methods, two of them refuted before use

I did not go straight to the answer. Two cheaper screens were tried first and both
are wrong **for this question**, each refuted against a case whose truth I already
knew. Recording them because they are the obvious things to try and a successor
will try them.

### Refuted 1 — match the filename against the folder name

The signal in folder 103 was that error files name themselves `舞-乙HiME`
(Mai-Otome) while the folder is `Mai-HiME`. Generalising that to "flag any error
file whose title does not match its folder" is wrong here, and measurably so.

Folders legitimately hold releases named under a different convention for the
**same work**:

| folder | the error file in it | same work? | why a name sweep fails |
| :--- | :--- | :--- | :--- |
| folder A | file A1 | yes | romanised JP title vs licensed EN title |
| folder A | file A2 | yes | same work again, CJK title + bracketed group token |
| folder B | file B1 | yes | romanised JP vs EN, plus a year in both |
| folder C | file C1 | yes | romanised JP vs a literal EN translation |
| folder D | file D1 | yes | romanised JP vs EN, subtitle dropped on one side |
| folder E | file E1 | yes | romanised JP vs EN, no shared token at all |
| folder F | file F1 | yes | word order inverted between the two renderings |

**REDACTED 2026-09-04.** This table previously carried seven folder titles and seven
media filenames, in three scripts, including a bracketed release-group token. **The
argument never needed them** — it is about the RELATION between two renderings of one
work, and the relation is what the right-hand column now states. Found by running
dev-3's retrospective auditor over my own zone; the names were the evidence for a claim
that is fully carried by their *kind*.

A name-based sweep flags every one of those. It would have produced a large,
confident, wrong number.

### Refuted 2 — compare durations

Cheap, content-derived, and still wrong. **Measured on folder 103**, the misfiled
files and the correctly named ones show the *same* delta against their master:

```
ep 23  file X (EN-titled)     1435.6 s  vs master 1564.9 s   -129.26 s  (8.26 %)
ep 23  file Y (CJK-titled)    1436.0 s  vs master 1564.9 s   -128.86 s  (8.23 %)
```

The correctly-named file is off by as much as the misfiled one. Duration mismatch
has several causes and "different programme" is only one of them — *different
cuts*, this campaign's largest family, produces it too. Duration cannot separate
them, so it is not the detector.

**Correction, on dev-1's measurement:** the figures in that block are CONTAINER
durations, and in Matroska the container duration is set by the longest stream —
usually video. On this master the container reads 1501.25 s while all three audio
streams read 1436.977 s, so the audio delta for id 237 is **+1.06 s, not
-63.21 s**. The argument above is unaffected (both files are compared against the
same master, so the two deltas move together), but every duration I quoted before
that correction was on the wrong axis. `audio_durations.py` re-probes on the audio
axis and records which of three sources answered, because Matroska usually leaves
the per-stream `duration` field empty and puts the value in a `DURATION` tag.

### Refuted 3 — sparse frame perceptual hashing, and this one I built

Twenty frames per file, DCT pHash, matched any-to-any so an offset would not
matter. Validated on id 6 (same content) and id 237 (different): **0.0 vs 24.0**,
a clean separation, thresholds written before the numbers.

Then I ran it and the first three rows all came back `candidate-different` — for
pairs that are plainly the same work under an English versus romaji title. I
killed the sweep.

**Why it failed, and it is this campaign's own lesson.** Twenty frames over 22
minutes is one every 66 seconds. With any offset at all, the samples on the two
sides never land on the same moment, so nothing matches and everything reads as
different. My positive control, id 6, was **frame-identical at zero offset** —
which is precisely the case that cannot expose the flaw.

*An armed guard that has never fired is indistinguishable from a working one*, and
a guard fired only on the easy case is indistinguishable from one that works. I
had two controls, both answers, a wide margin, and a useless instrument.

## The method actually used — chromaprint over every lag

Compare the audio the way the pipeline already does: raw chromaprint fingerprints,
cross-correlated at **every** lag, so an offset is found rather than tripped over.

```
peak = max over lags of (bit agreement between the two fingerprints)
       0 = chance, 1 = identical, computed over lags with >= 200 points of overlap
```

`fpcalc -raw -length 900` costs ~1.2 s per file, so the whole corpus is minutes,
not hours.

### Validated on controls whose truth was established by eye first

**This is the part that makes the number trustworthy**, and it is the step
refuted method 3 skipped by luck rather than by design.

| id | truth, established how | peak | instrument said |
| ---: | :--- | ---: | :--- |
| 6 | **same** — I extracted frames at 700 s from both and they are *identical* | 0.5635 | same-content ✅ |
| 125 | **same, WITH an offset** — durations differ by 11 s; frames at 650 s show the same warehouse, same red-lit windows, moments apart | 0.624 | same-content ✅ |
| 237 | **different** — frames at 600 s and 900 s are unrelated content, different setting and characters | 0.11 | candidate-different ✅ |

Id 125 is the control that matters: it is the exact case that killed method 3. The
instrument passes it with a wide margin. Thresholds (`same >= 0.30`,
`different < 0.15`) were written before any of these numbers were seen, and all
three land outside the ambiguous band.

### Known limitation, found by measurement, and what was done about it

`fpcalc` reads the **first** audio stream. On error id 349 that was a Japanese
original against an **English dub** on the master:

```
candidate : 1 stream  -- jpn
master    : 4 streams -- eng, jpn, jpn, jpn      <- fpcalc took eng
```

peak 0.0849 — flagged, for a pair whose titles say it is the same work. So a
second pass (`same_content_streams.py`) re-scores flagged pairs across **every**
stream combination and keeps the best.

On id 349 that pass returned **0.0932 on jpn→jpn** — still low. So the dub theory
does **not** explain that particular pair, and it stays a candidate for eye
verification rather than being explained away. Recording that because the tidy
outcome would have been to assume the fix worked.

## What a result from this does and does not mean

| peak | means |
| :--- | :--- |
| high | **conclusive that the two share content.** Different programmes do not correlate at any lag |
| low | a **candidate**. Music-only stretches, heavy re-encodes and differently mixed dubs also depress a peak. Verified by looking at frames before it is counted |
| none | the instrument could not run on that pair — **a third outcome**, not a quiet zero |

The headline number is therefore reported in two parts: candidates found, and
candidates **confirmed by eye**. They are not the same number and will not be
presented as one.

## Files

| file | what |
| :--- | :--- |
| `sweep_durations.py` | pairs every error to its master, probes both durations |
| `content_match.py` | the refuted pHash screen. **Kept deliberately** — a successor should see why it fails, not re-derive it |
| `same_content.py` | chromaprint, first stream, whole corpus |
| `same_content_streams.py` | chromaprint, every stream pair, for flagged rows |
| `runs/sweep-durations.json` | 315 rows, durations both sides |
| `runs/pass1-chromaprint.json` | 315 rows, peaks |

---

# RESULTS

**Answer: the misfiling is real, and it is NOT corpus-wide. 96 % of it is one
folder.**

| | entries | |
| :--- | ---: | :--- |
| folder 103 | **26** | three independent methods agree on the same 26 ids |
| the other **37 folders** | **1** | id 692, confirmed by eye |
| **total different-content** | **27** | **8.6 % of 315** |

So the headline 8 % survives, and the operational reading of it changes
completely: **it is one contaminated folder plus one stray file**, not a pattern
distributed across the corpus. Folder 103 alone holds 49 of the 315 error entries
and more than half of them are junk. Clean that folder and the family essentially
disappears.

## Why the folder-103 number is trustworthy — three methods, one answer

| method | by whom | evidence used |
| :--- | :--- | :--- |
| release names + **video frames at 600 s and 900 s** | me | id 237 confirmed by looking |
| **audio fidelity** in the pipeline's own `.log.error` | vmsam-forensic, independently | never read a filename or a frame |
| **delay-pattern shape** from the same logs | me, after the fact | classifies on pattern, not fidelity level |

The first two produced the **identical 26-id set** — `237, 238, 239, 241, 243,
245-265` — without either seeing the other's inputs. The third splits folder 103
as 23 `floor-scatter` + 3 `mixed` against 22 `step` + 1 `mixed`, i.e. the misfiled
files sit at the noise floor and the correctly-named ones show clean steps.

**The 23 correctly-named entries in folder 103 are genuine repair targets** — high
fidelity, clean plateaus. Only the other 26 are junk.

## The mistake this nearly made, and what stopped it

`floor-scatter` — every window at chance, delays scattered — looks exactly like
"no shared content". There are **45** such rows. Reporting them as misfiled would
have given ~14 %.

**18 of those 45 are PAL-ratio speed pairs.** Same content, drifting apart at
4.27 %, and precisely what `TASKS/008` exists to repair. dev-1 predicted this
before I measured it, citing `docs/AUDIO_SPEED_POLICY.MD`, which records
uncorrected PAL-against-film pairs at fidelity **0.558-0.562** — the same floor a
different-series pair sits on, because across a long window a 4.27 % rate
difference drifts the signals apart so no single shift aligns them. Verified at
the source.

**Fidelity level cannot separate "different content" from "speed mismatch."** The
separator is the duration RATIO, and the delay PATTERN — a speed relation drifts
monotonically, no-shared-content scatters and flips sign.

Had I reported the floor as misfiling I would have told dev-2 to stop repairing 18
files that are exactly its target.

## And my own screen produced false positives

Of the 4 floor-scatter rows outside folder 103 that were not speed candidates, I
checked all four by extracting frames at three timestamps and looking:

| id | chromaprint peak | eye verdict |
| ---: | ---: | :--- |
| 57 | 0.1025 | **same** — same scenes moments apart; one copy carries an `[adult swim]` bug |
| 121 | 0.1133 (jpn→jpn) | **same** — same characters, same art |
| 123 | 0.1454 | **same** — same series throughout |
| 692 | 0.1094 | **DIFFERENT** — a modern school setting against a fantasy costume series |

**Three of four were false positives**, including id 121 which scored low on a
matched jpn→jpn pair. My screen narrows candidates; it does not decide. Every
number above that says "different content" was confirmed by looking at frames.

Note id 692's own filename claims the same title as its master, so this one is not
detectable by name at all — the release is mislabelled at source. It is the reason
a content test was necessary rather than a naming heuristic.

## Corpus shape, from the pipeline's own recorded matrices

Extracted from the `.log.error` beside all 315 error files. **Read the caveats in
`extract_logs.py`**: these were written by an older revision, through the
integration loop, and the logged candidate path is sometimes historical.

| shape | n | % | meaning |
| :--- | ---: | ---: | :--- |
| `step` | 157 | 49.8 % | plateaus at high fidelity — **different cuts**, dev-1's 006 target |
| `mixed` | 51 | 16.2 % | unclassified by my heuristic |
| `monotone-drift` | 48 | 15.2 % | delays growing along the file — speed candidates |
| `floor-scatter` | 45 | 14.3 % | at chance — misfiled **or** PAL speed |
| `no-matrix` | 14 | 4.4 % | no stage-1 matrix in the log |
| *(of which)* `cmd_error` | 2 | | **instrument did not run** — ffmpeg failed. Never folded into a family |

PAL-ratio speed candidates overall: **30 (9.5 %)**, concentrated in folders 110
(11), 81 (9) and 172 (7).

These land near `CAMPAIGN.MD`'s campaign-1 shape — different cuts ~57 % against my
49.8 % `step`, speed 11 % against 9.5 %. **I am not claiming to have reproduced
campaign 1's classification**: my shape heuristic is mine, its buckets are not
theirs, and 16 % sits in `mixed` which their table has no equivalent for.

## What is still not established

- **25 of the 26** folder-103 files are attributed from the release name plus the
  contiguous run and the fidelity floor. Only id 237 was opened and looked at.
- `mixed` (51) and `no-matrix` (14) were not individually examined. A further
  different-content case could hide there — the 27 is a **floor, not a ceiling**.
- The 30 speed candidates are identified by **duration ratio only**. None has been
  shown to actually correct with `asetrate`.
- Whether the media behind the records changed since campaign 1 is untestable here:
  vmsam-forensic measured the id set as identical, but that is the set of RECORDS,
  and campaign 1's table carries no path or hash.

---

# CORRECTION, same day — the shape classifier had a defect, and the headline survived it

dev-1 found two faults in the classification above. Both are recorded here rather
than edited away, because the second one is a trap for anyone who reads these logs.

## Fault 1 — mine. Triples were flattened across track pairs

`extract_logs.py` pooled every `(fidelity, offset_points, delay_ms)` triple in a
log into one list, **across all track pairs**. So:

```
id 23   '0-0' = [0]   x10, fid 0.993-0.997
        '1-0' = [125] x10, fid 0.981-0.984
```

Each track pair is perfectly **constant**. The two values are the **master's own
two audio tracks 125 ms apart** — not a change point in the media. Flattened, it
read as `delays=[0, 125]` and my classifier called it a `step`. **I then
recommended those exact files as the best change-point exemplars.** A chimeric
repair driven off that would have repaired something that is not there.

`extract_logs2.py` parses the dict structure and separates **within-pair**
variation (a possible staircase) from **between-pair** variation (the master's own
track layout). It finds `between-pair-offset` for ids **20, 21, 22, 23** — exactly
the four dev-1 identified, reached independently.

**The corrected parse made the misfiling headline stronger.** Folder 103 now splits
**perfectly**: 26 `floor-scatter` and 23 `within-step` — every misfiled id at the
noise floor, every correctly-named one showing a step. Under the flattened parse it
was 23 + 3 `mixed`. **27 of 315 stands.**

| shape (track-pair aware) | n |
| :--- | ---: |
| `within-step` | 247 |
| `floor-scatter` | 50 |
| `no-matrix` | 12 |
| `between-pair-offset` | 4 |
| `cmd-error` — **instrument did not run** | 2 |

## Fault 2 — dev-1's measurement. The recorded delays are RESIDUALS

`first_delay_test` calls `recreate_files_for_delay_adjuster` (`mergeVideo.py:333`),
re-extracting the candidate shifted by stage 1's own answer. Everything after that
is measured against an **already-shifted** candidate. The shift is an arbitrary
millisecond value and almost never a whole multiple of the chromaprint hop
(`4096/3/11025` = 123.840 ms), and a fractional-hop shift misaligns the two
fingerprint grids.

dev-1's measurement on id 318 — same pair, only the shift changing:

| shift | hops | fidelity | residual |
| ---: | ---: | :--- | ---: |
| 0.000 ms | 0.000 | 0.968-0.975 | +552 |
| 495.360 ms | 4.000 | 0.967-0.974 | 0 |
| **552.000 ms** | **4.457** | **0.918-0.936** | **-138**  ← the pipeline's value |
| 619.200 ms | 5.000 | 0.967-0.974 | -138 |

Whole-hop shifts hold fidelity; the fractional one drops it into exactly the band
the log records. **So a one-point step in recorded residuals may be an artefact of
the re-extraction, and the recorded fidelities are depressed by the instrument
rather than by the media.**

That is dev-1's measurement on **4 files of 247** — two artefacts, one real, one
unreadable. **Four files is not a rate and is not carried as one.** What follows is
only that the recorded data *cannot distinguish* a one-point step from a shift
artefact.

### So steps are now ranked, not counted

| within-pair step | n | |
| :--- | ---: | :--- |
| **1 point only** | **20** | **suspect** — may be a shift artefact |
| 2 points | 17 | |
| 3-5 points | 23 | |
| >5 points | 187 | |

The suspect 20 are ids 5, 6, 7, 10, 11, 14, 15, 16, 17, 18, 19, 45, 71, 100, 172,
271, 316, 318, 319, 320. **Three independent corroborations that this bucket is the
right one:** dev-1's two measured artefacts (318, 319) and their unreadable case
(316) all land in it; and so does **id 6, which I confirmed frame-identical at
700 s by eye** — a genuinely aligned pair showing a one-point step is precisely the
predicted artefact. (271 is also in the bucket and dev-1 measured it as a *real*
step, so the bucket is suspect, not condemned.)

**Best change-point exemplars** — within-pair step ≥ 2 points, fidelity ≥ 0.90:
ids 90, 102, 104, 74 (three plateaus, `-250 → -125 → 0`), and 50, 62, 94 (two).
A three-plateau monotone descent at fidelity 0.91 is the least artefact-like shape
available: one fractional-hop shift cannot manufacture three ordered plateaus.

Caveat attached: all seven are folder 54 and mostly one series, so they are **not
seven independent observations** — closer to one series' encoding habit seen seven
times. Ids 224 and 143 (folder 103's correctly-named set, `[500, 750]`) offer a
different source.

## What this episode says about the numbers above

The corpus-shape table in RESULTS was computed with the flattened parser. Its
`step` figure of 157 was wrong in composition — 4 files were not steps at all, and
an unknown number of the one-point ones may be artefacts. **The misfiling answer
did not depend on it**: that rested on the fidelity floor, on three agreeing
methods, and on frames I looked at. A defect in the step classifier could not move
it, which is the one reassuring thing here.

---

# The pattern behind three separate mistakes in one day

Named here because it is more useful as a pattern than as three embarrassments,
and because between two agents we hit it three times before lunch.

**A fixture built by copying one file cannot test anything that depends on the two
files being different.**

| instance | whose | the control | why its pass was guaranteed |
| :--- | :--- | :--- | :--- |
| sparse-frame pHash screen | mine | id 6, same content | **frame-identical at zero offset** — the one case where sparse sampling cannot miss |
| clean-edge change-point estimator | dev-1's | a file against a copy of itself with a cut | it tested for correlation reaching **1.00000**, which only identical content can do. On a real pair of different encodes, aligned content reads 0.93–0.96 and the hypotheses were not separable. Refuted on the first real file |
| gap-start bisection | dev-1's | reversed `lo`/`hi` | returned its own input bounds; caught because the number was implausible, not because the code complained |

In all three the control passed for a reason supplied by the construction rather
than by the thing under test. `AGENT.MD`'s rule — *an armed guard that has never
fired is indistinguishable from a working one* — has a sharper corollary:

> **A guard fired only on the easy case is indistinguishable from one that works.
> Ask what property of the fixture guarantees the pass, and whether the real input
> has that property.**

## The cleanest instance of the shift artefact in this corpus — id 6

Joint, and neither half is worth much alone:

| | |
| :--- | :--- |
| ground truth (mine) | **frame-identical at 700 s**, established by extracting both frames and looking |
| duration delta | **exactly 0.000 s** |
| what the log records | a one-point step |
| absolute measurement (dev-1's) | **−125 ms constant on all ten windows**, fid 0.940–0.949 |

A pair independently known to be aligned, recorded as stepping. That is dev-1's
re-extraction artefact demonstrated on a file whose alignment was established by a
method with no shared code path — the eye.

## Corpus facts established while answering dev-1

- **No high-fidelity staircase exists outside folders 54 and 103.** `step ≥ 2
  points AND fidelity ≥ 0.90`, excluding those two folders, returns **zero rows**.
  A fact about the corpus, not a gap in the filter. Third-origin exemplars must
  come from a lower fidelity band (42 rows at ≥ 0.80 across 14 folders).
- **Two track pairs stepping together is the strongest available corroboration**
  that a step is real: id 267 shows `'0-0'` and `'0-1'` both `[1000, 1250]`. A
  fractional-hop artefact would have to hit two track pairs identically to fake it.
- **The point count is portable across window lengths; the millisecond value is
  not** (dev-1, measured: the same physical offsets read 774/516 ms at a 60 s
  window, 810/540 at 30 s, 900/600 at 15 s — six points and four points every
  time). My screen discards the lag entirely and reports only a peak value, so no
  number of mine is affected; recorded so nobody later compares one against a
  pipeline delay.

## An open disagreement, deliberately not resolved by assertion

dev-1 and I recover the correlation quantum with slightly different totals:
**271/27/3** (mine) against **269/27/3 plus 2 flagged** (theirs). The arithmetic
269 + 2 = 271 is tidy, and their proposed cause — two files whose track pairs carry
125 and 126 — **does not hold in my data**: both named files have a single track
pair, and every window in each gives exactly 125.0.

My guess, offered as a guess: we recover it by different routes. I take
`-delay_ms / offset_points`, which recovers the value the pipeline **used**; a
prediction from `int(lengthFile/n*1000)` can land 1 ms low on a boundary case. That
would split *between methods* on boundary files, not *between track pairs* within
one file.

**Written up as: both methods agree on 299 of 301 and differ by 1 ms on two, cause
not established.** Not as a mechanism neither of us has confirmed.

---

# The PAL confusion, hit three times by three agents

**No correlation-based instrument can separate "different content" from "speed
mismatch."** A speed relation destroys correlation *by construction*, so a
genuinely-repairable PAL pair reads exactly like unrelated media. Three agents hit
this in one day with three different instruments.

| who | instrument | what it said | what was true |
| :--- | :--- | :--- | :--- |
| the pipeline | stage-1 fidelity | 0.558–0.562 floor | `docs/AUDIO_SPEED_POLICY.MD` records this for **uncorrected PAL**, the same floor unrelated content sits on |
| me | full-file chromaprint peak | 18 of my 45 floor rows "different" | **PAL speed pairs** — caught only because dev-1 warned me before I published |
| the forensic agent | 60 s chunk prominence, 12 probes | folder 110's 11 files, 0/12, "no shared audio on two independent languages" | **PAL speed pairs — verified visually** |

## Folder 110, settled

The folder alternates strictly, and the ratio is the tell:

```
id 46 ep1  err 1270.7  mst 1324.0  ratio 1.041956   <- flagged "no shared audio"
id 47 ep1  err 1321.5  mst 1324.0  ratio 1.001889
id 51 ep2  err 1267.3  mst 1321.5  ratio 1.042748   <- flagged
id 52 ep2  err 1318.5  mst 1321.5  ratio 1.002281
```

All 11 flagged files: **1.0420–1.0431**. PAL/film is **25/23.976 = 1.042709**.
Every one within 0.001. The "runs ~53 s shorter" observation offered as evidence of
missing content *is* the arithmetic of a 4.27 % speedup: 1324 s played fast is
1270.4 s.

**Verified by a prediction stated before looking:** if this is a speed relation of
the same content, candidate at `t` must show the same frame as master at
`t × 1.042709`, and master at a naive `t` must show something else.

| id 46, t = 300 s | |
| :--- | :--- |
| candidate(300.0) | two characters against a wall of red numerals |
| master(312.8, **scaled**) | **the same frame** — same shot, same poses |
| master(300.0, naive) | a different shot entirely |

id 91 at 250 s and 450 s: unmistakably the same series and episode.

**Why a 60 s chunk cannot survive it:** at 4.27 %, a 60 s chunk accumulates 2.56 s
of internal drift, so it aligns with *no* position in the master — it is stretched
relative to all of them. And a two-language control does not rescue it: a French
dub and an English dub of a PAL-sped episode are *both* sped, so both read as
absent against a film-rate master. Running two languages tests for a dub, not for
a speed relation.

## The rule this produces

> **Compute the duration ratio before concluding "different content" from any
> correlation measure.** It costs one `ffprobe` per file and separates the two
> families that every correlation instrument conflates.

Had this not been checked, the corpus figure would have been **~14 %** instead of
8.6 %, and **11 files that task 008 exists to repair would have been written off as
junk.**

**What is not claimed:** that these 11 actually correct with `asetrate` — that is
task 008's to establish. Only that the content is shared and the ratio is PAL. Two
of the eleven were verified visually; the other nine are attributed from the ratio
and the strict alternation.

## The rule, corrected — it needs both directions

My published version was one-directional and would have mislabelled in the other
direction. The forensic agent supplied the missing half; both are needed.

> 1. **A collapsed correlation does not imply different content.** A speed relation
>    destroys correlation by construction — check the duration ratio.
> 2. **A speed-like ratio does not imply a speed relation.** Content differences
>    (trimmed openings, different edits) change duration too — check whether the
>    correlation *actually collapsed*.
>
> **Call it a speed pair only when the ratio is within ~0.001 of the rate AND the
> correlation collapsed.**

### My own count was over-inclusive, and this is the correction

I reported **30** PAL candidates at a tolerance of 0.004. Applying the guard:
**19**, exactly the forensic agent's set, reached independently.

The 11 I lose all fail the second test — their correlation never collapsed:

| ids | ratio | |ratio − PAL| | fidelity max | verdict |
| :--- | ---: | ---: | ---: | :--- |
| 229–244 (one folder, 9 files) | 1.0457–1.0458 | 0.0031 | **0.950–0.974** | genuine *different cuts* |
| 26, 353 | 0.9553–0.9555 | 0.0872 | 0.834, 0.974 | inverse-ratio false positives of my looser band |

The nine-file cluster is instructive: they sit tightly at a *consistent* 1.0457,
which looks like a rate — but they correlate at 0.95+. A genuine 4.6 % rate
difference could not do that. So their duration difference is **content**, not
rate: that folder is Blu-ray against streaming, where opening and ending lengths
differ. A tight ratio cluster is not by itself a speed signature.

### The forensic agent's spread test — an independent check on my 26

A speed relation is **one rate**, so a speed population **clusters**:

| population | ratio spread |
| :--- | ---: |
| folder 110 (PAL, 11 files) | **0.00127** |
| folder 103 (my misfiled 26) | **0.0458** — 36× wider |

Folder 103 is not a speed population, and could not be. **My 26 now carry a fourth
independent check**, and it is one I did not think to make.

### Reconciliation

The forensic agent's corrected corpus figure is **27 (8.6 %)** — my number, by a
different route, from audio fidelity and ratio rather than from names and frames.

Their scattered set was **101, 121, 299, 692**; my guess that it matched my
57/121/123/692 was **wrong**. Two of theirs turned out PAL, one was my visual
same-content call, and only 692 survives. Recorded because guessing at another
agent's set and being close is exactly how a wrong number gets adopted by both
sides.

---

# CORRECTION — two of my visual verifications were wrong, and a third family exists

I gave the forensic agent nine visual verifications as ground truth. **Two were
wrong**, both in the same way, and they challenged them rather than adopting them.

## What I did wrong

For id 57 I wrote: *"the same scenes moments apart, an elevator interior in both,
the same purple-haired character against the same neon background."* **Scene-level.
Unambiguous.**

For id 123 I wrote: *"same fantasy setting and character designs across three
timestamps."* **Series-level** — which holds for *any* episode of that show. I gave
the two observations the same weight, and both came out of my mouth as the word
"same".

Re-tested properly, extracting the candidate at three timestamps and the master at
**both** offsets the other agent's audio had matched at (`+30.1 s` and `-1.9 s`):

| t | candidate | master (either offset) |
| ---: | :--- | :--- |
| 400 s | blonde girl with glasses, close-up | mansion interior / night exterior |
| 700 s | same character, dark scene | two characters outdoors / a stage |
| 1000 s | pointed ears, facial scars | costumed group / blonde close-up |

**Not one scene matches at either alignment.** Same series, different episode.

id 121 is the same story: four timestamps, a 35 s delta so same-`t` frames should be
near-adjacent if it were one episode, and every pair shows the same characters in a
**different scene**.

## The third family

Both are the same folder, episodes 11 and 12. That is not a coincidence:

| family | what is wrong | count |
| :--- | :--- | ---: |
| folder 103 | wrong **series** filed into the folder | 26 |
| id 692 | wrong **series**, and its own filename lies | 1 |
| folder 40 (ids 121, 123) | **right series, wrong EPISODE paired** | 2 verified |

**"Same series, different episode" is invisible to every test either of us built.**
The names match. The series matches. The character designs match. Only the scenes
and the dialogue audio disagree — and the other agent's *theme-only* signature
(shared OP/ED with zero dialogue correlation) is the sharpest detector anyone has
for it, because that is exactly what two episodes of one series look like.

Operationally these belong with folder 103, not with weak correlation: no constant
offset exists, no repair can succeed, and a repair that appeared to succeed would
splice **the wrong episode** into the library. *Weak correlation* implies same
content that is hard to correlate, which is precisely what these are not.

**How many more exist is unmeasured.** Two are verified; nothing has swept for the
pattern.

## Corrected ground-truth set

| verdict | ids |
| :--- | :--- |
| same content, **scene-level** verified | 6, 46, 57, 91, 125 |
| different **series** | 237, 692 |
| **wrong episode**, same series | 121, 123 — *were listed as "same". Withdrawn.* |

The other agent's classifier scores **9 of 9** against the corrected set, not 7 of 9.
Both "conflicts" were mine.

## The lesson, and it is the one instrument failure today that was my own judgement

The other three failures were code. This one was me applying a weaker test and not
noticing, because **both strengths of evidence produce the same word.**

> **Record a verdict's evidence LEVEL beside its conclusion, not just the
> conclusion.** "Same series" and "same scene" are different claims; written down as
> "same", they are indistinguishable a day later — including to the person who wrote
> them.

Had I labelled each verification `scene-level` or `series-level` when I made it, the
weak ones would have been visible without anyone challenging them.

---

# id 110 — a file that is not damaged, not misfiled, not desynced, and still refused

The forensic agent's sharpest open case, and it turns out to be a **metadata**
defect, not a synchronisation one.

**1. The video is frame-identical at zero delay.** Candidate and master at the same
timestamps `t = 200, 500, 800, 1100 s`: the same frames in all four pairs — a map
card, a seated figure in a hall, a character with a staff against a castle wall, a
close-up. **Scene-level**, labelled as such. Audio durations agree to **13 ms**
(1472.000 vs 1472.013). The correct delay is exactly **0**.

**2. Why it refused — my first guess was wrong.** I suspected the pipeline compared
`eng` against `jpn` by index. Checking the code first: it selects a common language
and compares tracks of that language on both sides
(`videoObj.audios[common_language_use_for_generate_delay]`). Both files carry `jpn`.
It should have compared like with like.

**3. The tags lie.** Chromaprint over every stream pair:

| pair | peak |
| :--- | ---: |
| candidate[eng] vs master[jpn] | 0.2845 |
| candidate[jpn] vs master[jpn] | 0.2845 |
| **candidate[eng] vs candidate[jpn]** | **1.0000** |

The candidate carries **one audio track duplicated under two language labels**.
There is no genuine Japanese audio in it. The pipeline correctly selected `jpn` on
both sides and thereby compared an English dub against a Japanese original — no
confident peak, spurious ~218 s delays, refusal.

0.2845 is exactly right for that: well above chance (~0.10 here) and well below
same-content (0.56–0.62), because a dub and its original share the music-and-effects
bed and nothing else. The other agent's 4-of-12 probes at delay 0.0 are the same
fact — the probes landing on music matched, those on dialogue did not.

## Why this one matters more than its count

`CAMPAIGN.MD`'s *no shared language* row is defined as *"no audio language in
common, **or the tags lie**"*. So this is not a new family — it is the **second half
of an existing one**, and the two halves have opposite remedies:

| half | remedy |
| :--- | :--- |
| genuinely no shared language | unfixable |
| **the tags lie** | **trivially mergeable at delay 0** — if anything could establish it |

This file is not damaged, not misfiled and not desynced. Its video aligns perfectly.
It is refused because its metadata is wrong, and the correct answer is unreachable
through audio correlation **by construction**.

## A cheap detector, offered and not swept

> **If two audio streams within ONE file are identical (fingerprint 1.0000 or equal
> MD5) but carry different language tags, at least one tag is false.**

A single-file test needing no master, and the merge engine already does MD5 audio
grouping, so the primitive exists. **One file measured. No count claimed.**

---

# My "no-matrix" bucket held FIVE outcomes, and one of them is an engine finding

The forensic agent corrected me on one file: I called it *"no stage-1 matrix — that
is 'instrument did not run' shaped"*. **The instrument ran.** Its log carries a
`first_delay_test` event with ten delays at mean fidelity 0.5726; only the
`Multiple delay found` dump is absent, because the refusal exited via the stage-2
disagreement path. Verified myself rather than relayed.

That is exactly the axis `BRIEF_COMMON` says gets lost — *"I could not measure"* and
*"I measured and the answer was bad"* are different answers — and my classifier had
pooled them. Splitting all 14 files in that bucket by what their log actually says:

| outcome | n | |
| :--- | ---: | :--- |
| **no common language** | **7** | a *correct refusal*, nothing to measure. Matches the other agent's count for that row exactly |
| **stage-2 variance refusal** | 4 | measured, and see below |
| **`ffmpeg` cmd error** | 2 | the instrument genuinely did not run |
| stage-1 ran, no dump (different exit) | 1 | measured, answer was bad |

Five outcomes where I had reported one.

## The engine finding: a single outlier window refuses an excellent measurement

Nine files corpus-wide carry a stage-2 variance refusal. **Four of them look like
this:**

| id | windows agreeing | their spread | median | outlier |
| ---: | :--- | ---: | ---: | ---: |
| 12 | 9 of 10 | **2.18 ms** | −56.7 ms | +788 ms |
| 13 | 9 of 10 | **1.56 ms** | −19.7 ms | +814 ms |
| 45 | 8 of 10 | **1.16 ms** | +90.7 ms | — |
| 108 | 9 of 10 | **2.11 ms** | +48.7 ms | −454 ms |

Nine windows agreeing to **two milliseconds** is an *excellent* measurement — far
better than the frame quantum of 41.7 ms. These files are refused because
`statistics.variance` is computed over the **whole** set including the outlier, and
one bad window dominates it.

The code even notices: the recorded message reads *"…but the first and last part
have a similar delay"*.

**Where it lives:** `mergeVideo.py:582–594`, **outside every tagged zone** — frozen.
So this is a finding for the owner, not an edit.

### A units nuance, offered carefully

`max_delay_variance_second_method = 0.004` is compared against
`statistics.variance(...)`, which returns **seconds squared**. So the implied
tolerance on *spread* is √0.004 ≈ **63 ms of standard deviation**, not 4 ms.
`AGENT.MD` describes it as *"0.004 s"*, which reads as a 4 ms tolerance.

Both readings agree that these four files should pass on their nine good windows
(σ ≈ 0.7 ms either way) and that the outlier is what refuses them, so **no
conclusion here depends on the nuance** — but the figure is quoted in a frozen
document and is worth stating precisely.

### What I am NOT claiming

That these four files would merge **correctly**. Their recorded measurement looks
excellent; that is not the same as being right, and `AGENT.MD` is explicit that a
refusal is a correct outcome. What is measured: nine windows agree to ~2 ms and the
file is refused on one outlier. Whether a robust statistic *should* accept them is a
ruling for whoever owns stage 2 — and it would need a real merge to settle, not a
log.

---

# Stage-2 variance refusals are THREE things, not one

dev-1 proposed a testable hypothesis: that stage-1's one-point residual steps and
stage-2's wild outliers are the same bad window seen by two instruments — quantised
in stage 1 (so it lands one point away) and unquantised in stage 2 (so it lands
hundreds of milliseconds away). They asked me to check whether the outlier sits at
the **same window index** as the stage-1 step, since I had both matrices parsed.

**It splits.** Across all nine files carrying a stage-2 variance refusal:

| group | n | pattern |
| :--- | ---: | :--- |
| **A** real change point | 4 | stage-1 step and stage-2 outliers at the **same contiguous tail block** |
| **B** first-window only | 3 | outlier at **index 0**, and stage 1 recorded **nothing at all** |
| **C** degenerate | 2 | more than half the values flagged |

## B refutes the hypothesis — one instrument saw nothing

```
id  12  [788.2, -56.5, -56.8, -57.2, -56.2, -57.3, -57.7, -57.2, -56.2, -55.5]  outlier idx [0]
id  13  [813.8, -19.5, -19.0, -20.0, -19.3, -20.4, -20.6, -19.6, -19.8, -20.1]  outlier idx [0]
id 108  [-453.9, 47.8, 47.8, 47.8, 47.8, 49.8, 49.9, 49.9, 49.6, 49.6]          outlier idx [0]
```

These logs carry **no stage-1 matrix at all** — stage 1 was content and the refusal
came wholly from stage 2. So it cannot be one window seen by two instruments: the
other instrument saw nothing.

And it is **systematic, not random**: all three single-outlier files have it at
**index 0**. The natural suspects are what sits at the start of a file — a logo,
black, silence, a different intro — with stage 2's unquantised FFT locking onto an
arbitrary lag where stage 1's quantised chromaprint did not. The outlier signs
differ (+788, +814, −454), so it is not a fixed offset.

**Two hedges.** Three files is not a rate. And I am assuming index 0 is the
**earliest window in time** — the windows come from
`generate_begin_and_length_by_segment` and **I have not confirmed the ordering is
chronological.** If it is not, the "first window" reading collapses to "one bad
window" after all. Cheap for someone to check.

## A supports the mechanism but means something else

```
id 100   stage-1 steps [6,7,8,9]   stage-2 outliers [6,7,8,9]   EXACT
id 172   stage-1 steps [6,7,8,9]   stage-2 outliers [6,7,8,9]   EXACT
id  71   stage-1 steps [6,7,8,9]   stage-2 outliers   [7,8,9]   SUBSET
id  45   stage-1 steps       [6]   stage-2 outliers     [6,9]   index 6 shared
```

The matching indices are a **contiguous block of the last four windows in both
instruments**. That is not two instruments tripping over one bad window — it is two
independent instruments observing a **real change point** in the last 40 % of the
file and agreeing on where it is. Both are correct, and it is a useful
cross-validation of stage 1 against stage 2.

## What survives

dev-1's consequence — *a file refused by stage 2 may need nothing more than one
window dropped* — **holds, for group B only**, which is precisely the group where
the remaining nine windows agree to 1.5–2.2 ms. Narrower than first handed over, and
better defined.

## The units finding, in dev-1's stronger form

They verified mine and extended it: **one constant is compared against two
dimensionally different quantities sixteen lines apart.**

| line | comparison | units |
| :--- | :--- | :--- |
| `mergeVideo.py:582` | `variance(list_delay) < 0.004` | **s²** → implies σ ≈ 63.2 ms |
| `mergeVideo.py:584` | `abs(first - last) < 0.004` | **s** → 4 ms flat |
| `mergeVideo.py:594` | `variance(delay_detected) > 0.004` | **s²** |

`AGENT.MD`'s *"max_delay_variance_second_method = 0.004 s"* matches **584** and not
582/594 — so the document describes one of the two uses rather than being wrong. The
dimensional analysis is dev-1's; I had only spotted the s² side.

---

# Family B closed: three early-step files, ground truth from four instruments

What I first called *"nine windows agreeing to 2 ms, refused by one bad window"* and
then corrected to *"real steps"* is now measured end to end.

| id | change point | step | instruments agreeing |
| ---: | :--- | ---: | ---: |
| 108 | **146–148 s** | **501 ms** | 4 — my bisection, dev-1's locator, forensic's probe grid, the pipeline's own log |
| 12 | **158–160 s** | **840–875 ms** | 4 — same, plus the direct FFT difference |
| 13 | *(not bisected)* | ~750 ms | 2 — dev-1 and forensic |

**id 12, 8 s probes:**

```
[148,156] +577.6  corr 0.8990   PRE
[152,160] +567.0  corr 0.9215   PRE
[156,164] +577.6  corr 0.2673   spans the boundary  <- correlation collapses
[160,168] +1412.4 corr 0.7483   POST
[164,172] +1423.0 corr 0.8317   POST
```

The boundary-spanning probe collapsing to 0.267 while its neighbours sit at 0.89–0.92
is the same signature id 108 showed — independent confirmation that the transition is
**sharp**, not gradual.

## A loose end resolved, and it was an instrument bias not a second step

dev-1 flagged that their short-window reading of id 12's post-step region (12 points
= 1500 ms) disagreed with the ten-window scan (11 points = 1375 ms), and named two
candidates: a second change point near 220–230 s, or a bias in their short windows.
They flagged it rather than picking one.

Probed straight across it and well beyond:

```
[160,200] +1412.4   [220,240] +1423.0   [400,460] +1412.4
[180,220] +1423.0   [230,250] +1417.6   [700,760] +1423.0
[200,220] +1412.4   [240,260] +1412.4
```

**Every probe from 160 s to 760 s reads +1412 to +1428 ms** — 16 ms of spread across
ten minutes, at correlations 0.86–0.97. **No second step.** The direct measurement of
their disputed region is ~1417 ms = 11.3 points, so their ten-window reading of 11 was
right and their short-window absolute was 83 ms high.

**Their own rule protected them:** the *location* was identical at both window lengths
and the *step in points* was 7 at both. The bias is in the absolute value only — which
is exactly what they had already told another agent never to exchange across window
lengths.

## Why this set matters more than the three files

All three sit inside the head blind spot, so **dev-1's committed locator returns
`kind="constant"` on every one of them — confidently wrong.**

That makes them the best acceptance test available for the change-point work,
*because* the current code fails on them and **the expected answers exist
independently of the thing being tested.**

> A constructed fixture — a file against a copy of itself — has an expected answer
> **you supplied**. This set has one **you did not**. That is the fixture trap of this
> morning stated as its own cure rather than as a warning.
