# 006 — Locate the change points inside a staircase

**Agent:** `vmsam-dev-1` (measurement). **Pairs with 007.** **State:** open.
**Campaign objective 2, the measurement half.**

## The signal

A window sequence reading

```
(0.92, -1, 125)  (0.93, 0, 0)  (0.90, -1, 125)
```

is **one content with a small insertion or deletion**: correlation stays high
throughout while the offset steps by whole points. One point = **125 ms**, the
chromaprint frame — every delay in this corpus is a multiple of it.

That is the shape. `first_delay_test` sees the disagreement across windows and
refuses the pair, which is correct: no single constant offset exists. It records
what it saw in `video_obj.delayFirstMethodAbort[best_video.filePath]` →
`[1|2, delays]`, filled at 12 points. **You do not need to re-run the correlation
that just failed** (`SPEC_ZONE_A.MD` §1).

## Goal

Build the method that turns that staircase into **change points**: where, in
master-timeline seconds, does the offset step, and by how much. Windows are 12,
their length is twice their spacing so consecutive ones overlap by half — a step
is bracketed to roughly one spacing by the window data alone. Getting from there
to a frame is the part that needs an instrument, 005's.

## Target-first — binding, `AGENT.MD` and `CAMPAIGN.MD` both

1. **Build one correct answer by hand first**, on one real error file, with
   whatever tools you like. The located change point, in seconds, and how you
   know. This exists **before** the code.
2. **State what result would refute you, before the run.** A prediction written
   afterwards is not a prediction.
3. Only then write code, and check it reproduces the hand answer.

## The rule that decides whether this counts

**Your work counts when dev-2 calls it, not when it passes a test.** Five pieces
of campaign-1 code were written, reviewed and unit-tested green and never called
by anything. Agree the interface with dev-2 **before** you build behind it, and
pair this with 007 before either is declared done.

## Ask the Lead for

- a tagged call site, if you need one — one agent at a time, one edit
- a container run, through `vmsam-ci` — never directly
## What changed

| file | what |
| :--- | :--- |
| `src/change_point_locator.py` | new module. Full-file probe grid, plateau grouping, clean-probe transition bracketing, per-stream offsets, bounds clamp. Reads no configuration |

Called by `vmsam-dev-2`'s repair. Returns numbers, never files.

## What was measured

**The sign convention holds.** `candidate_time_ms = master_time_ms + candidate_offset_ms`,
derived from `audioCorrelation`'s algebra and confirmed on a constructed 1000 ms
deletion (read -1000) and independently by dev-2 from the opposite direction.

**Plateau offsets are confirmed against outside instruments.** On error id 108,
-950.68 / -1453.83 / -951.47 against `vmsam-ci`'s and dev-2's -950.0 / -1453.9 /
-952.9 — agreement to ~1.5 ms.

**On error id 266, committed code, no hand plan or patch** (dev-2's run): repaired,
2 audio and 21 subtitle tracks, 0 declined, 0 failed, both tracks aligned to worst
1.12 ms and 1.00 ms over six probes at r 0.961-0.995, `VMSAM_FABRICATED=chimeric`
in the final mkv. Locator log: 35 probes over [0,1428] s, median fidelity 0.969,
zero sign flips, five plateaus.

**All branches fired both ways on real files**: id 237 declines (fidelity 0.576,
31 distinct, 18 flips); id 318 returns `constant`; id 147 returns two change
points. The drop-vs-decline branch fired with literals in both directions.

**Transitions: ONE independent check only.** dev-2's 8 s / 2 s profile of id 266
puts one transition in (628, 632) s, containing my (629.0, 631.5). One transition,
one instrument, one file, and a coarser step — favourable, **not confirmation**.

## What was dropped

- **The frame-level video refiner.** dev-2's repair fills the bracket from the
  master, correct by definition, at 0.49 % of an episode against 15 %. An
  instrument with no consumer.
- **`_align_and_find_gap`** (`frame_compare`): aggregates cost only where a
  reference frame matched, so inserted content is invisible to its argmax.
- **`get_cut_time.py`** as a starting point: divides `second_correlation`'s
  already-in-seconds offset by the frame rate at all four call sites.
- **Reading `delayFirstMethodAbort`**: those are residuals, and a fractional-hop
  re-extraction manufactures one-point steps.
- **The clean-edge estimator and the majority-model bisection**, both refuted.

## What is blocked

Nothing is blocked. Two known defects, both mine, neither fixed yet and both
deliberate:

1. **Transition brackets are unconfirmed at scale.** The only bracket numbers I
   produced before the current method came from a version later found 35x
   over-claimed. One outside check exists.
2. **Bound-only brackets overlap.** `_bracket_transition`'s fallback returns
   `region_end + PROBE_WINDOW_SECONDS`, which can push past the next run's first
   probe and swallow a plateau. On id 266 this dropped 2 segments and left 26.6 %
   of the timeline filled from the master — waste, not damage. Not fixed while
   dev-2 validates against this exact block; changing it mid-validation would make
   any disagreement unattributable.

Narrative, measurement logs and the full history of what was refuted:
`AGENT_NOTES.MD` and `VMSAM_HELP_AI/dev-1/lab/`.


---

## The offsets have a reference frame, and now they say so

`_probe` extracts with `ffmpeg -ss t -i file`, which seeks by **presentation
timestamp** and so absorbs each stream's container `start_time`. A consumer reading
a stream decoded from its beginning measures **sample position**. The two differ by
exactly the master-minus-candidate `start_time` difference.

**Measured on error id 144:**

    MASTER      idx 2 eac3 eng  start_time 0.048     idx 5 aac jpn  start_time 1.103
    CANDIDATE   idx 1 aac  eng  start_time 0.000     idx 5 aac jpn  start_time 0.120

    eng   probe  +14.85 ms   decode-from-zero   +66.35 ms   differ   51.5   (start_time  48)
    jpn   probe  +20.32 ms   decode-from-zero +1002.81 ms   differ  982.5   (start_time 983)

`vmsam-dev-2`'s verifier measured the produced file's ja track at **-1002.75 ms**,
matching the decode-from-zero figure to 0.06 ms — **so the produced file behaves like
sample position while the plan was expressed in presentation timestamps.** A frame
mismatch across the join, not a wrong number on either side.

**Corpus scan, ffprobe only, no decode — 315 pairs:**

| band | files | consequence |
| :--- | ---: | :--- |
| any mismatch | **112** (36 %) | |
| **>= 100 ms** | **25** | above the repair's tolerance: declines, visible. **24 of 25 are one release** |
| **20 - 100 ms** | **77** | **under the tolerance: would ship silently**, median 21 ms, max 84 ms |
| < 20 ms | 10 | |

The 25 visible ones explain `vmsam-dev-2`'s four release-32 declines and predict
twenty more from the same release. **The 77 are the ones that matter**, and they are
the same shape as id 266's 27.5 ms per-stream split: real, wrong, and under every
tolerance in the chain.

**The files this campaign has repaired and measured on are clean**: ids 46, 266, 108
and 237 all read 0.0 ms, so nothing already claimed rests on this.

### What changed, and what deliberately did not

**Changed:** the plan now carries `offset_reference_frame: "presentation_timestamp"`.
Both framings are defensible; an implicit one is not — the same number meant two
things depending on who read it. **A key, not a convention.**

**Not changed: the numbers.** Making the probe ignore `start_time` would silently
alter every offset this module has ever emitted, including plans already repaired
against and the id 46 plan that verified at 1.125 ms. **Which framing the assembly
wants is the repair's to say**, and changing the meaning of a shipping field on a
guess is how a wrong number gets confirmed by everything downstream.

### A hypothesis tested and reverted

I was confident the cause was inaccurate input seeking — `-ss` before `-i` landing on
a packet boundary. I implemented a two-stage accurate seek, re-ran the identical
test, and **got identical numbers to two decimal places**. Reverted.

**A defensible improvement with no evidence behind it is not a fix.** This campaign
has already shipped five pieces of code that were reviewed green and never called; a
sixth that was reviewed green and repaired nothing would be worse, because it would
look like the repair.

### Why this explanation is believed

**It predicted a pattern it was not built to explain.** ids 144 and 157 decline at
1103.4 and 1103.8 ms — agreeing to 0.4 ms — while their change points sit 330 s
apart. Every episode of that release carries the same jpn `start_time`, so the same
983 ms enters each identically. **An explanation that also accounts for a regularity
you were not trying to account for is a different class of evidence from one that
fits the case in front of you.**


---

## Correction: there was no second frame, and one field survives

The section above says the plan carries `offset_reference_frame` because a
presentation-timestamp probe and a sample-position consumer measure different
quantities. **`vmsam-dev-2` tested that directly rather than inferring it — PTS seek
against the `atrim` its assembler actually uses, same file, same stream, same
window, on a stream with a 120 ms `start_time`: 0.0 ms apart over 20 s.** Same frame.
There was never a conversion to do between the plan and the assembly.

**So the key is withdrawn**, along with the master-minus-candidate difference emitted
beside it. A key naming a distinction that does not exist is not hygiene: it is a
second thing to keep true and a second thing to get wrong.

**One field survives, because it measures something that predicts a failure.**
`master_reference_start_time_ms` — the master reference stream's own `start_time`. A
consumer that rebuilds a track starting at PTS 0 is misaligned from the master's
track by exactly this amount, and dev-2's four release-32 declines match it to under
5 ms across three distinct values:

    id 144   decline 1103.4 ms   master jpn start_time 1103.0   apart 0.4 ms
    id 157           1103.8                            1103.0         0.8
    id 174           1059.4                            1055.0         4.4
    id 225            887.6                             887.0         0.6

### The corpus, re-sliced on the quantity that matters

The first scan counted master-minus-candidate mismatches. **That was the wrong
column.** Re-sliced on the master's own `start_time`, 315 pairs, ffprobe only:

| band | files | consequence |
| :--- | ---: | :--- |
| master `start_time` nonzero on a shared language | **65** (21 %) | |
| >= 100 ms | **25** | visible: declines. **24 of 25 are one release** |
| 20 - 100 ms | **33** | under a 100 ms tolerance — would not be noticed |
| < 20 ms | 7 | |

The earlier figures of 112 and 77 were mismatches and are superseded.

### And "identical" was too strong — my own scan says so

I wrote that *every episode of release 32 carries the same jpn `start_time`, so the
same 983 ms enters each identically*. **The 24 episodes carry ten distinct values
spanning 216 ms:**

    887 892 892 892 911 959 959 983 983 983 983 988 1031 1055
    1079 1079 1079 1079 1103 1103 1103 1103 1103 1103

I had this data when I wrote the sentence. **Two of the four files dev-2 declined
sit at 1103 and the other two at 1055 and 887** — so the errors did not agree to
0.4 ms across the release; they agreed across the pair that shares a value, and
tracked the input on the pair that does not.

**That is a better test than the one I claimed.** An identical input predicting an
identical error fits any constant. **A clustered input predicting a clustered error
that tracks it file by file, across three distinct values, is the evidence** — and
it was available the whole time in the numbers I had already measured.


---

## A probe before the master stream begins reads a spurious offset

**Found by `vmsam-dev-2` asking a question I could not answer without looking**: its
two defect terms are separable per file only when the plan begins at master 0,
because otherwise it emits a head master-fill piece whose own term equals the
verifier's. **The plan's first-segment start is my output**, so which files are
decidable was mine to determine.

**Two of the four I checked did not begin at 0, and the reason was mine.**

`_probe` seeks by presentation timestamp. On a stream whose first packet is stamped
1.103 s, a probe at t=0 cannot return audio from t=0 — there is none — so it returns
the stream's opening against a candidate window that really does start at 0, and
reports the `start_time` rather than the relation being measured.

    [change_point_locator] segment 0 unusable (offset 1003 ms, master [0.0,0.0]);
                           dropped, not declining

The head probe read +1003 ms against a body of +22, the run splitter read that as a
change point at the very start, and the zero-width first segment was dropped — so
the plan began at master **100000 ms** instead of 0.

**The cure: start the grid where the reference stream actually begins.** Everything
before that is a region no probe can measure.

### Fired both ways on real media

| id | lang | master start_time | plan start BEFORE | plan start AFTER |
| ---: | :--- | ---: | ---: | ---: |
| 144 | jpn | 1103 ms | **100000 ms**, segment dropped | **0**, nothing dropped |
| 375 | jpn | 74 ms | **100000 ms**, segment dropped | **0**, nothing dropped |
| 144 | eng | 48 ms | 0 | 0 |
| 173 | jpn | 42 ms | 0 | 0 |

**The two that were wrong are fixed and the two that were right are unchanged.**

### What it does downstream

- **Removes 100 s of master fill** on each affected file — that content is now
  candidate audio.
- **Makes dev-2's two defect terms separable on those files**, because there is no
  head master piece for the assembler term to act on.
- **Changes emitted offsets only where the master reference stream has a nonzero
  `start_time`** — 65 of 315 pairs. Files at 0.0, including id 46 and id 266, are
  untouched, so nothing already repaired or claimed moves.

### Why this one was worth making and the seek change was not

Both were plausible. **This one has two files that change and two that do not**, a
log line naming the mechanism, and a predicted downstream consequence that was
independently wanted. The accurate-seek change had a good argument and produced
identical numbers to two decimal places — so it was reverted. **The difference is
not confidence, it is whether anything measurable moved.**
