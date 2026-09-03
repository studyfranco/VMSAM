# 008 — A third speed relation: measure the coefficient, do not assume it

**Agent:** `vmsam-dev-1` (measurement). **Pairs with 009.** **State:** open.
**Campaign objective 3, the measurement half.**

## The signal

```
(0.58, 65, -8125)   (0.61, 382, -47750)
```

Low fidelity **throughout**, and an offset that **grows along the file**. That is
a speed relation, not a bad measurement — and the low fidelity is a consequence of
the drift inside each window, not evidence against the reading.

Check the arithmetic yourself before trusting the example: 65 points × 125 ms =
8125 ms, 382 × 125 = 47750 ms. Both delays are exact multiples of the chromaprint
frame, which is what a quantised measurement of a continuous drift looks like.

## Goal

**Measure the coefficient. Do not assume PAL's 1.042709.** Two windows give it as
the change in delay over the change in position. Report what you measured, with
its uncertainty — the delay is quantised to 125 ms, so a two-window estimate has a
floor you can state exactly, and more windows lower it.

Then say **which relation it is**, or that it is none you recognise. `1000/1001`,
`24/23.976`, `25/24`, `30/29.97` and their inverses are all candidates; so is
"nothing standard", and that is a legitimate answer.

## What the campaign wants from it

`CAMPAIGN.MD`: treat it **exactly as PAL is treated — same method, different
coefficient. Audio and subtitles are both taken.** `docs/AUDIO_SPEED_POLICY.MD`
rules `asetrate` for PAL and NTSC and explains why the *right* answer inverts if
the source was already pitch-corrected at origin. **Read the inverting case before
proposing a method for a third relation** — the ruling is about the physics of how
the speed change was introduced, not about the factor.

That document is **frozen**. A third relation confirmed by measurement **extends**
it, and reaches the owner as a finding in `AGENT_NOTES.MD`, never as an edit.

## Expect both at once

Insertions and deletions can sit **on top of** a speed relation. A residual
staircase after removing the slope is 006's problem, not a refutation of the
slope. Say which you are looking at.

## Target-first — binding

Build one correct answer by hand, on one real error file, before writing code.
**State the coefficient you expect and what result would refute you, before the
run.** Campaign 1 has the counter-example: a method tuned until its spread was the
tightest of a whole benchmark, 0.008 ms, while it correlated the wrong language and
was wrong by 21 ms.

## What changed / measured / dropped / blocked

**State: coefficient measured, verdict reached, both on two real error files.
Not yet applied — that is dev-2's 009 and it closes when a track exists.**

### Measured — the coefficient, and it is NOT PAL

`vmsam-forensic` named ids 33 and 31 (folder 164, eng+fre, eac3). Their ten
recorded delays are **monotone decreasing** (id 33: 3968 -> 2108 ms) with a gcd of
**124 ms**, which matches my duration->quantum model for their 418 s / 454 s
windows. Regressing delay against window centre:

| id | slope | `speed_ratio` = master/candidate | ~95 % |
| ---: | ---: | ---: | :--- |
| 33 | -0.992431 ms/s | **1.000993** | [1.000953, 1.001034] |
| 31 | -0.993192 ms/s | **1.000994** | [1.000954, 1.001034] |

**Residual scatter about the fit is 38.3 and 41.0 ms against 124/sqrt(12) = 35.8 ms
predicted by pure quantisation.** The residuals are therefore almost entirely the
quantum, the linear model is not strained, and **no staircase is hiding under the
slope** — which is the thing this task warns about.

**The relation is `1001/1000`.** It lies inside both intervals; PAL's 1.042709 is
excluded by four orders of magnitude. **`1001/1000` and `24/23.976` are the same
relation, not two candidates** — 23.976 rounds 24000/1001, and 24/(24000/1001) =
1001/1000 exactly. This confirms `vmsam-forensic`'s NTSC finding on two files, and
I carry its hedge: 2.5 % of the corpus is a **floor**, not a count, because a
PAL-rate file drifts ~9 s inside a 214 s window, collapses to weak correlation, and
is structurally invisible to the method that found these.

### Measured — the verdict, by the policy's own method

`docs/AUDIO_SPEED_POLICY.MD` rules that the verdict comes from applying both
corrections and fingerprinting each against the master, at a 0.85 bar, with three
outcomes and a decline. Four arms, 1200 s window (0.099 % needs room: 0.41 s of
drift over the pipeline's 418 s, 1.19 s over 1200 s):

| arm | id 33 | id 31 |
| :--- | ---: | ---: |
| uncorrected (control) | 0.7411 | 0.7374 |
| **asetrate CORRECT** | **0.9588** | **0.9551** |
| asetrate WRONG (sign control) | 0.6569 | 0.6562 |
| rubberband | 0.9484 | 0.9453 |

**Verdict: `asetrate`**, both files, clearing the bar.

**The sign control is the point.** Weakness 3 of the policy records a sweep that
used `* r` instead of `/ r`, landed twice as wrong, and had rubberband win 33 of 33
with nothing internal able to catch it. Here the wrong-direction arm lands **0.084
below the uncorrected control** — it makes things worse, as a doubled error must.
So the direction handed to dev-2 is demonstrated, not asserted.

### Finding for the owner — the detector's margin does not carry from PAL to NTSC

`AUDIO_SPEED_POLICY.MD` measures rubberband paying a **~0.109** processing handicap
and losing by 0.278-0.313 against real masters. **Measured here the gap is 0.0104
and 0.0098** — an order of magnitude smaller.

Consistent with the document rather than against it: it already records that
against a real master `asetrate` reaches only 0.94-0.98 *"because real pairs carry
content and codec differences no correction removes."* Both arms press against that
same ceiling, which compresses the distance between them. At 4.27 % the losing arm
had far to fall; at 0.099 % it does not.

**The ruling stands** — `asetrate` wins on both files, and its justification is
physical (the exact inverse of the transfer) rather than statistical. **The limit
is that the margin which would detect an inverting case in this corpus is ~0.010,
not ~0.109**, so a detector built to the document's expected separation could miss
one. Two files agreeing in direction and magnitude is evidence the margin is real;
it is not a rate, and I am not reporting it as one.

`AUDIO_SPEED_POLICY.MD` is frozen. This **extends** it by measurement and reaches
the owner as a finding in `AGENT_NOTES.MD`, never as an edit.

### What dev-2 gets

`verdict: "asetrate"`, `speed_ratio: 1001/1000` **as the exact rational**, not my
point estimate — my measurement is consistent with it and *less precise than it*,
so propagating my noise into a file would be worse than using the known relation.
`ratio_precision: 3 decimals (1.001)`; the interval is +/-4e-5, so the fourth
decimal is already uncertain. dev-2 tags the factor it **applied** to full
precision (`SPEC_ZONE_A.MD` §4, correct as written) and carries my interval in the
repair report, where it can say what the tag cannot.

### Strengthened to all eight files, from forensic's final table

`forensic/data/classified_v2.json` classes **8** files as `speed mismatch`, all
`family_v2 = "speed mismatch (NTSC)"`, **all folder 164**. Regressing delay on
window centre, with the uncertainty floor taken from the **quantum** (`q/sqrt(12)`)
rather than from the residuals alone — the floor my own quantum finding implies:

| id | slope ms/s | ratio | resid sd | q-floor |
| ---: | ---: | ---: | ---: | ---: |
| 26 | -1.022438 | 1.001023 | 39.8 | 35.8 |
| 27 | -1.014954 | 1.001016 | 55.2 | 35.8 |
| 29 | -0.981684 | 1.000983 | 35.5 | 35.8 |
| 31 | -0.993192 | 1.000994 | 41.0 | 35.8 |
| 33 | -0.992431 | 1.000993 | 38.3 | 35.8 |
| 35 | -1.015700 | 1.001017 | 39.2 | 35.8 |
| 37 | -0.984921 | 1.000986 | 38.3 | 35.8 |
| 113 | -0.995384 | 1.000996 | 38.0 | 35.8 |

**Mean 1.001001, 95 % [1.000990, 1.001012].** `1001/1000` lands essentially dead
centre; PAL's 1.042709 is excluded by four orders of magnitude.

**Two caveats that travel with the number.** All eight are **one series**, so eight
files are not eight independent observations — the family is one show's encoding
habit seen eight times. And the **between-file scatter is 2.8x the pooled standard
error**, so the eight disagree by more than quantisation explains: the honest
interval is the scatter-based `[1.000990, 1.001012]`, not the tighter pooled one.
Within each file the residual sd sits at or just above the 35.8 ms quantisation
floor, so quantisation dominates there, as expected.

### CORRECTION after the promotion — the corpus is NOT NTSC-only

`vmsam-forensic` has corrected its own earlier finding: there are **19 PAL files**
in the corpus, roughly **twice** the NTSC population. Everything measured above
covers the **8-file NTSC subfamily only**, and the mean of 1.001001 is that
subfamily's coefficient, not the corpus's.

**The hedge carried with the original result is what turned out to be load-bearing**
and it is restated rather than quietly dropped: *2.5 % is a floor, not a count,
because a PAL-rate file drifts ~9 s inside a 214 s window, its correlation
collapses, and the method that found the NTSC eight is structurally unable to see
one.* PAL files were hiding exactly where that hedge said they would be.

**A stated limit of my own instrument.** forensic reports the 19 carry no usable
delay measurement — a 4.27 % relation destroys correlation, so stage 1, forensic's
prominence measure and **`change_point_locator` alike find nothing on them**. The
duration ratio is the only evidence. My locator will decline them, and that decline
is correct rather than a gap: it cannot measure what does not correlate. dev-2's
speed path handles them from the duration ratio, with a synthesised full-coverage
segment, because demanding segments would make the intended case look like failure.

So the honest scope of this task's number: **`1001/1000`, `asetrate`, for the NTSC
family. PAL is a separate population, larger, and measured by a different route.**

### Blocked

Nothing. The measurement half of objective 3 is done on two files. It closes when
dev-2's 009 produces a resampled track that its post-mux verifier accepts.
